

export GMX_MAXBACKUP=-1

#CREATE outputs for all the molecules using a forloop

#NEED: directory with all pdb files and directory with all mol2 files
#how to get these? via an website and some code

#for i in {001..010} #to run molecule 1 to 10
#todo= {006,042,076}
# todo2= {006,078,095,125,128,129,150,171,176,183,186,218,223,229,233,237,238,244,248,255,260,262,263,264,265,294,297,321,334,337,341,361,383,385,391,392,407,411,412,444,460,462,466,467,474,485,492,496,508,511,517,523,530,533,538,548,556,565,569,587,588,600,602,603,615}

#echo "Fixed using array:"
#a=(001 002 003)
#for x in "${a[@]}"

echo go
#./RunMDsimulations.sh >> testing.log 2>&1
# for i in {003,006,015,016,043,052,054,055,056,062,070,080,086,094,109,115,199,203,220,224,225,229,230,246,248,249,250,255,259,275,279,291,292,307,309,320,338,353,365,370,374,376,377,392,395,420,426,428,436,443,449,453,478,480,498,511,514,517,519,520,523,538,539,542,551,563,566,579,580,581,586,588,601,602,609,621,623,624,628,629,636,637,641,644,649,658,668,669,680,689,702,705,706,707,711,721,722,724,728,729,733,734,739,746,750,754,759,769,781,782,783,784,788,794,811,815,834,845,852,855}
for i in {015,016,115,629,782,783}
do
    echo %%%%%%%%%%%%%%%%%         NEXT MOLECULE         %%%%%%%%%%%%%%%%%%%%%%%%%%%
    #specify the molecule/files
    nozeroi=$(echo $i | sed 's/^0*//')
    echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    echo ${nozeroi}
    LIGPDB=$i.pdb
    LIGMOL2=$i.mol2
    LIGNAME=$i

    #create the setup
    mkdir -p setup${LIGNAME}
    cd setup${LIGNAME}

    #change mol and pdb file names and paste them in setup folder
    nozeroi=$(echo $i | sed 's/^0*//') #remove forward zero padding
    [ -f ../mol/$i.mol2 ] && mv ../mol/$i.mol2 ../mol/$i.mol2 #
    [ -f ../pdb/$i.pdb ] && mv ../pdb/$i.pdb ../pdb/$i.pdb #
    cp ../pdb/$LIGPDB ../mol/$LIGMOL2 .

    echo $LIGMOL2

    sed -i 's/.*\/\([0-9]\{3\}\)\.pdb/\1/' ${LIGMOL2} #replaces docked line with the molecule number
    
    #2. prepare ligand forcefield parameter and topology file
    $SILCSBIODIR/cgenff/cgenff_to_gmx.sh mol=$LIGMOL2

    LIGTOP=${LIGMOL2%.*}_gmx.top #the topology file. We assign a variable name to the files created
    LIGPDB=${LIGMOL2%.*}_gmx.pdb
    cp posre.itp posre_ALL_${LIGNAME}.itp
    
    ########ignore############
    #not being used
    # 3. prepare protein topology file
    GMXPDB=${LIGPDB%.*}_lig.pdb
    GMXTOP=${LIGPDB%.*}_lig.top

    gmx pdb2gmx -f $LIGPDB -o $GMXPDB -p $GMXTOP -ff charmm36 -water tip3p
    cp posre.itp posre_HEAVY_${LIGNAME}.itp
    ##########################

    #determine how many atoms you want to restrain
    #RESTR=ALL
    RESTR=HEAVY_${LIGNAME}

    sed -i "s/posre.itp/posre_$RESTR.itp/g" ${LIGTOP} #change the position restraints in the file AAA_gmx.top 
    
    #5. solvate system
    MARGIN=1.5 #nm
    BOXPDB=${LIGPDB%.*}_box.pdb #LIGPDB = 1_gmx.pdb
    SOLTOP=${LIGPDB%.*}_solvated.top
    SOLPDB=${LIGPDB%.*}_solvated.pdb
    cp $LIGTOP $SOLTOP

    gmx editconf -f ${LIGPDB} -o ${BOXPDB} -d ${MARGIN}
    gmx solvate -cp ${BOXPDB} -cs -o ${BOXPDB} -p ${SOLTOP} #no water solvent model selected, we use default TIP3P
    #ERROR in grompp: No default proper dih. types

    # 6. neutralize system
    gmx grompp -f ${SILCSBIODIR}/data/gromacs/gmx_inputs/simple.mdp -c ${BOXPDB} -p ${SOLTOP} -o ions${LIGNAME}.tpr
    #echo 15 | gmx genion -s ions.tpr -o ${BOXPDB} -p ${SOLTOP} -pname NA -nname CL -neutral #maybe 13?
    gmx genion -s ions${LIGNAME}.tpr -o ${BOXPDB} -p ${SOLTOP} -pname NA -nname CL -neutral #maybe 13?
    
    #echo $CRYST > $SOLPDB #puts var $cryst in SOLPDB
    cat $BOXPDB >> $SOLPDB
    
    #create outputfiles for every molecule
    mkdir -p ../alloutputsFinal/$LIGNAME
    cd ../alloutputsFinal/$LIGNAME

    mv ../../setup${LIGNAME}/charmm36.ff .
    mv ../../setup${LIGNAME}/${SOLTOP} .  
    mv ../../setup${LIGNAME}/${SOLPDB} .
    mv ../../setup${LIGNAME}/posre_ALL_${LIGNAME}.itp .
    mv ../../setup${LIGNAME}/posre_HEAVY_${LIGNAME}.itp .
    mv ../../setup${LIGNAME}/ions${LIGNAME}.tpr .
    mv ../../setup${LIGNAME} .
    
    # 7. run MD simulation
    PREFIX=${LIGNAME%.*}_min_steep
    MINPDBsteep=${PREFIX}.pdb #pdb of ligand with minimization done
    gmx grompp -f ${SILCSBIODIR}/data/gromacs/gmx_inputs/emin1steep.mdp -c ${SOLPDB} -p ${SOLTOP} -o ${PREFIX} -r ${SOLPDB} 
    gmx mdrun -deffnm ${PREFIX} -c ${MINPDBsteep}

    #echo 12 0| gmx energy -f ${PREFIX}.edr -o potentialsteep.xvg
    #xmgrace potentialsteep.xvg

    PREFIX=${LIGNAME%.*}_min_cg
    MINPDB=${PREFIX}.pdb
    gmx grompp -f ${SILCSBIODIR}/data/gromacs/gmx_inputs/emin2cg.mdp -c ${MINPDBsteep} -p ${SOLTOP} -o ${PREFIX} -r ${MINPDBsteep} 
    gmx mdrun -deffnm ${PREFIX} -c ${MINPDB}

    #11 or 12 depending on having improper-dih.
    #echo 12 0| gmx energy -f ${PREFIX}.edr -o potentialcg.xvg #not for every ligand the same due to some not having the option improper-dih
    #xmgrace potentialcg.xvg

    #equilibration
    PREFIX=${LIGNAME%.*}_equil
    EQUILPDB=${PREFIX}.pdb
    gmx grompp -f ${SILCSBIODIR}/data/gromacs/gmx_inputs/equil_posre.mdp -c ${MINPDB} -p ${SOLTOP} -o ${PREFIX} -r ${MINPDB} #SOLPDB
    gmx mdrun -deffnm ${PREFIX} -c ${EQUILPDB}

    #production
    PREFIX=${LIGNAME%.*}_prod
    PRODPDB=${PREFIX}.pdb
    gmx grompp -f ${SILCSBIODIR}/data/gromacs/gmx_inputs/prod.mdp -c ${EQUILPDB} -p ${SOLTOP} -o ${PREFIX} -r ${EQUILPDB}
    gmx mdrun -deffnm ${PREFIX} -c ${PRODPDB}

    cd ../..
done
